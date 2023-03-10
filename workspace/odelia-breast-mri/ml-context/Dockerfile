FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install           \
   build-essential python3-dev python3-pip     \
   python3-setuptools --no-install-recommends -y
RUN conda install pip

RUN pip3 install --upgrade pip protobuf && pip3 install \
   matplotlib pandas simpleitk pydicom pandas numpy tqdm openpyxl

RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
RUN pip3 install \
  torchio torchaudio monai pytorch-lightning tensorboard scikit-learn matplotlib seaborn
RUN pwd
RUN mkdir -p /tmp/hpe-swarmcli-pkg
RUN mkdir -p /tmp/channel/
COPY swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl /tmp/hpe-swarmcli-pkg/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl
 #COPY environment.yaml /tmp/hpe-swarmcli-pkg/environment.yaml
 #COPY setup.py /tmp/hpe-swarmcli-pkg/setup.py
 #RUN conda env update --file /tmp/hpe-swarmcli-pkg/environment.yaml --prune
 #COPY environment.yaml /tmp/hpe-swarmcli-pkg/environment.yaml
COPY host1-partial-data /tmp/test
COPY Clinical_and_Other_Features.xlsx /tmp
COPY mnist.npz /platform/data/
RUN pip3 install /tmp/hpe-swarmcli-pkg/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl
#RUN pip3 install -e ./tmp/hpe-swarmcli-pkg/setup.py