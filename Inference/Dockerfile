FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output/images/mediastinal-lymph-node-segmentation/ \
    && chown -R user:user /opt/app /input /output

WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN apt-get -y update
RUN apt-get -y install git

USER user

RUN python -m pip install --user -U pip
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user -rrequirements.txt
RUN python -m pip install git+https://github.com/MIC-DKFZ/nnUNet.git

RUN python -m venv /opt/app/venv --system-site-packages
RUN . /opt/app/venv/bin/activate && pip install TotalSegmentator==1.5.7
COPY --chown=user:user download_weights_totalsegm.py /opt/app/
RUN . /opt/app/venv/bin/activate && python /opt/app/download_weights_totalsegm.py

ADD --chown=user:user ./weights /opt/app/weights
ADD --chown=user:user ./nnUnet /opt/app/nnUnet
ADD --chown=user:user ./preregistration /opt/app/preregistration
COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user preprocessing_functions.py /opt/app/
COPY --chown=user:user post_processing_functions.py /opt/app/
COPY --chown=user:user predictor.py /opt/app/

ENTRYPOINT [ "python3", "-m", "process" ]
#ENTRYPOINT [ "bash" ]
