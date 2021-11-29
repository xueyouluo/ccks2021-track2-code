FROM nvcr.io/nvidia/tensorflow:19.10-py3

# set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# install tzdata & curl package
RUN apt update && apt-get install -y tzdata wget curl

RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
&& dpkg-reconfigure -f noninteractive tzdata

# pretained models and datas
COPY user_data/electra /user_data/electra
COPY user_data/extra_data /user_data/extra_data
# COPY user_data/track3 /user_data/track3

# add code
COPY Dockerfile /Dockerfile

COPY code /code

WORKDIR /code
CMD ["sh","run.sh"]