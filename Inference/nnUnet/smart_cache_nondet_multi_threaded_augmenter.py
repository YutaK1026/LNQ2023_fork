# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
import math
from copy import deepcopy
from typing import List, Union
import threading
from builtins import range
from multiprocessing import Process
from multiprocessing import Queue
from queue import Queue as thrQueue
import numpy as np
import logging
from multiprocessing import Event
from time import sleep, time

from batchgenerators.dataloading.data_loader import DataLoader
from threadpoolctl import threadpool_limits

try:
    import torch
except ImportError:
    torch = None


def pin_memory_of_all_eligible_items_in_dict(result_dict):
    for k in result_dict.keys():
        if isinstance(result_dict[k], torch.Tensor):
            result_dict[k] = result_dict[k].pin_memory()
    return result_dict


def fill_batch_queue(in_idx_queue: thrQueue, in_cached_data: list, transform, out_queue: thrQueue,
                     abort_event: Event, abort_batch_event: Event,
                     pin_memory: bool, worker_list: List[Process],
                 gpu: Union[int, None] = None, wait_time: float = 0.02):
    do_pin_memory = torch is not None and pin_memory and gpu is not None and torch.cuda.is_available()

    if do_pin_memory:
        print('using pin_memory on device', gpu)
        torch.cuda.set_device(gpu)

    item = None

    while True:
        try:
            if abort_event.is_set():
                return
            # check if this thread should be ended
            if abort_batch_event.is_set():
                print("End batch thread!")
                return

            # check if all workers are still alive
            if not all([i.is_alive() for i in worker_list]):
                abort_event.set()
                raise RuntimeError("One or more background workers are no longer alive. Exiting. Please check the "
                                   "print statements above for the actual error message")
            if item is None:
                # abort if all data idx are processed
                if in_idx_queue.empty():
                    sleep(wait_time)
                    continue
                else:
                    item_idx = in_idx_queue.get()
                    # in_idx_queue.task_done()
                    assert item_idx < len(in_cached_data)
                    item = in_cached_data[item_idx]
                    if transform is not None:
                        item = transform(**item)
                    if do_pin_memory:
                        item = pin_memory_of_all_eligible_items_in_dict(item)

            # we only arrive here if item is not None. Now put item in to the out_queue
            if not out_queue.full():
                out_queue.put(item)
                item = None
            else:
                sleep(wait_time)
                continue

        except Exception as e:
            abort_event.set()
            raise e


def fill_replace_queue(queue: Queue, data_loader, transform, thread_id: int, seed,
                       abort_event: Event, wait_time: float = 0.02):
    # the producer will set the abort event if something happens
    with threadpool_limits(1, None):
        np.random.seed(seed)
        data_loader.set_thread_id(thread_id)
        item = None

        try:
            while True:

                if abort_event.is_set():
                    return
                else:
                    if item is None:
                        item = next(data_loader)
                        if transform is not None:
                            item = transform(**item)

                    if abort_event.is_set():
                        return

                    if not queue.full():
                        queue.put(item)
                        item = None
                    else:
                        sleep(wait_time)

        except KeyboardInterrupt:
            abort_event.set()
            return

        except Exception as e:
            print("Exception in background worker %d:\n" % thread_id, e)
            traceback.print_exc()
            abort_event.set()
            return


class SmartCacheNonDetMultiThreadedAugmenter(object):
    """
    Non-deterministic but potentially faster than MultiThreadedAugmenter and uses less RAM. Also less complicated.
    This one only has one queue through which the communication with background workers happens, meaning that there
    can be a race condition to it (and thus a nondeterministic ordering of batches). The advantage of this approach is
    that we will never run into the issue where everything needs to wait for worker X to finish its work.
    Also this approach requires less RAM because we do not need to have some number of cached batches per worker and
    now use a global pool of caches batches that is shared among all workers.
    THIS MTA ONLY WORKS WITH DATALOADER THAT RETURN INFINITE RANDOM SAMPLES! So if you are using DataLoader, make sure
    to set infinite=True.
    Seeding this is not recommended :-)
    """

    def __init__(self, data_loader, transform, num_processes, cache_size, replace_rate=0.1, num_batch_cached=2,
                 seeds=None, pin_memory=False, wait_time=0.02):
        logging.debug("Initialize SmartCacheNonDetMultiThreadedAugmenter()")
        self.pin_memory = pin_memory
        self.transform = transform

        if isinstance(data_loader, DataLoader): assert data_loader.infinite, "Only use DataLoader instances that" \
                                                                             " have infinite=True"
        self.generator = data_loader
        self.num_processes = num_processes
        self.num_batch_threads = num_batch_cached

        # necessary queues and chaches
        self.cache_size = cache_size
        self._replace_queue: Queue = None
        self._cached_data = [None for _ in range(self.cache_size)]
        self._cache_idx_queue: thrQueue = None
        self._replace_num: int = math.ceil(self.cache_size * replace_rate)
        self._replace_start_idx = 0
        self._batch_queue_size = num_batch_cached
        self._batch_queue: thrQueue = None

        #  multiprocessing
        self._replace_processes = []
        self._batch_fill_threads = []
        self.abort_event = None  # aborts everything
        self.abort_batch_event = None  # aborts batch threads
        self.initialized = False

        self.wait_time = wait_time

        if seeds is not None:
            assert len(seeds) == num_processes
        else:
            seeds = [None] * num_processes
        self.seeds = seeds

    def _fill_cache(self, indices=None) -> list:
        """
        Compute and fill the cache content from data source.

        Args:
            indices: target indices in the `self.data` source to compute cache.
                if None, use the first `cache_num` items.
        """
        assert self._cached_data is not None and len(self._cached_data) == self.cache_size
        if indices is None:
            indices = list(range(self.cache_size))
        num_replaced = 0
        while num_replaced < len(indices):
            if not self._replace_queue.empty():
                item = self._replace_queue.get()
                idx = indices[num_replaced]
                self._cached_data[idx] = item
                print(f"{idx}", end=',', flush=True)
                num_replaced += 1
            else:
                sleep(self.wait_time)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __get_next_item(self):
        item = None

        while item is None:
            #
            if self.abort_event.is_set():
                # self.communication_thread handles checking for dead workers and will set the abort event if necessary
                self._finish()
                raise RuntimeError("One or more background workers are no longer alive. Exiting. Please check the "
                                   "print statements above for the actual error message")

            if not self._batch_queue.empty():
                item = self._batch_queue.get()
                self._batch_queue.task_done()
                # print('+',end='', flush=True)
            else:
                if not any([thread.is_alive() for thread in self._batch_fill_threads]):
                    raise RuntimeError("All threads for batch queue are finished but still batches are expected!\n"
                                       "Possibly iterations per epoch are larger than cache-size.")
                # print('.',end='', flush=True)
                sleep(self.wait_time)
        return item

    def __next__(self):
        if not self.initialized:
            self.start()

        item = self.__get_next_item()
        return item

    def start(self):
        if not self.initialized:
            self.finish()
            logging.debug("Start SmartCacheNonDetMultiThreadedAugmenter()")

            self._replace_queue = Queue(self._replace_num)
            self._batch_queue = thrQueue(self._batch_queue_size)
            self.abort_event = Event()
            self.abort_batch_event = Event()

            logging.debug("starting workers")
            if isinstance(self.generator, DataLoader):
                self.generator.was_initialized = False

            # START processes of the refill queue
            print(f"Start {self.num_processes} loader processes.")
            for i in range(self.num_processes):
                self._replace_processes.append(Process(target=fill_replace_queue, args=(
                    self._replace_queue, self.generator, None, i, self.seeds[i], self.abort_event, self.wait_time
                )))
                self._replace_processes[-1].daemon = True
            _ = [i.start() for i in self._replace_processes]

            # FILL Cache from replace queue until is completely full  # is it possible more effciently?
            print(f"Fill cache of size {self.cache_size} / replace queue size is {self._replace_num} ....", end="", flush=True)
            self._fill_cache(indices=list(range(self.cache_size)))
            self._replace_start_idx = 0
            print("Done.")

            # start threads to perform data augmentation and push into batch queue
            self._start_batch_threads()

            self.initialized = True
        else:
            logging.debug("MultiThreadedGenerator Warning: start() has been called but workers are already running")

    def _start_batch_threads(self):
        #for thread in self._batch_fill_threads:
        #    if thread.is_alive():
        #        print('ERROR: still batch threads alive!')
        #        self.abort_batch_event.set()
        #        sleep(self.wait_time)
        #        self.abort_batch_event.clear()  # if was set before
        #del self._batch_fill_threads
        #self._batch_fill_threads = []

        #if self._cache_idx_queue is not None and not self._cache_idx_queue.empty():
        #    print('ERROR: Still batches in queue!')
        #    while not self._cache_idx_queue.empty():
        #        _ = self._cache_idx_queue.get()

        gpu = torch.cuda.current_device() if torch is not None and torch.cuda.is_available() else None

        # silly but maybe works, fill a queue with all idxs in the cache and let threads consume the idxs
        if self._cache_idx_queue is None:
             self._cache_idx_queue = thrQueue(self.cache_size + self._batch_queue_size)
        for idx in range(self.cache_size):  # add cache at beginning of queue
            self._cache_idx_queue.put(idx)
        # Todo: bit hacky will not work if queue size is large and replace-num is large
        for i in range(self._batch_queue_size):
            if self._cache_idx_queue.full():
                break
            idx = (self._replace_start_idx + self._replace_num + i) % self.cache_size  # add elements that are certainly not replaced
            self._cache_idx_queue.put(idx)

        if len(self._batch_fill_threads) == 0:  # at beginning
            print(f"Start {self.num_batch_threads} threads for data augmentation.")
            # def fill_batch_queue(in_idx_queue: thrQueue, in_cached_data: list, transform, out_queue: thrQueue,
            #                     abort_event: Event, abort_batch_event: Event,
            #                     pin_memory: bool, worker_list: List[Process],
            #                     gpu: Union[int, None] = None, wait_time: float = 0.02):
            for t in range(self.num_batch_threads):
                bt = threading.Thread(target=fill_batch_queue,
                                      args=(self._cache_idx_queue, self._cached_data, self.transform, self._batch_queue,
                                            self.abort_event, self.abort_batch_event,
                                            self.pin_memory, self._replace_processes, gpu,
                                            self.wait_time)
                                      )
                bt.daemon = True
                bt.start()
                self._batch_fill_threads.append(bt)
        count = sum([1 for thr in self._batch_fill_threads if thr.is_alive()])
        print(f"{count} threads for data augmentation alive.")

    def update_cache(self):
        if not self.initialized:
            return
        replace_idx_list = list(range(self._replace_start_idx, self._replace_start_idx + self._replace_num))
        replace_idx_list = [idx % self.cache_size for idx in replace_idx_list]
        print("Replace in cache: ", end='')
        self._fill_cache(replace_idx_list)
        print(".")
        self._replace_start_idx = (self._replace_start_idx + self._replace_num) % self.cache_size
        # print(f"\nReplaced: {replace_idx_list}")
        self._start_batch_threads()

    def finish(self):
        if self.initialized:
            self.abort_event.set()
            sleep(self.wait_time * 2)
            [i.terminate() for i in self._replace_processes if i.is_alive()]

        del self._replace_queue, self._cached_data, self._batch_queue, self._replace_processes, \
            self._batch_fill_threads, self.abort_event, self.abort_batch_event

        self._replace_queue, self._cached_data, self._batch_queue, \
            self.abort_event, self.abort_batch_event = None, None, None, None, None
        self._replace_processes = []
        self._batch_fill_threads = []
        self._cached_data = [None for _ in range(self.cache_size)]

        self.initialized = False

    def restart(self):
        self.finish()
        self.start()

    def __len__(self):
        return self.cache_size

    def __del__(self):
        logging.debug("MultiThreadedGenerator: destructor was called")
        self.finish()


if __name__ == '__main__':
    from tests.test_DataLoader import DummyDataLoader
    from scipy import stats
    num_items = 123
    dl = DummyDataLoader(deepcopy(list(range(num_items))), 2, 3, None,
                         return_incomplete=False, shuffle=True,
                         infinite=True)
    #data_loader, transform, num_processes, cache_size, replace_rate=0.1, num_cached=2, seeds=None,
    #             pin_memory=False, wait_time=0.02):
    mt = SmartCacheNonDetMultiThreadedAugmenter(data_loader=dl, transform=None, num_processes=4, cache_size=99,
                                                replace_rate=0.3, num_batch_cached=5, seeds=None, wait_time=0.02)
    #mt.start()

    counter = [0] * num_items
    values = []
    st = time()
    for epoch in range(5):
        for i in range(len(mt)):
            b = next(mt)
            #print(epoch, i, b, end="/")
            for idx in b:
                counter[idx] += 1
                values.append(idx)
        mt.update_cache()
    end = time()
    print(end - st)

    mt.finish()
    print(counter)
    #values = list(range(num_items))
    stat_result = stats.kstest(values, stats.uniform(loc=0.0, scale=float(num_items)).cdf)
    print(stat_result)
