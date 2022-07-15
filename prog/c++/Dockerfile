FROM ubuntu:20.04
#DEBIAN_FRONTEND=noninteractive のときは、インタラクティブな設定をしなくなる，自動インストールの際には便利．
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y build-essential cmake clang libssl-dev vim git
WORKDIR work