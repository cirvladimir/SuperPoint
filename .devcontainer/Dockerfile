FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

RUN apt update && apt install -y openssh-client libgl1

RUN pip3 install opencv-contrib-python==4.7.0.72 tqdm==4.65.0

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo "$SNIPPET" >> "/root/.bashrc"
