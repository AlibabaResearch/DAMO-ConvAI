FROM nvcr.io/nvidia/pytorch:21.02-py3
USER root
WORKDIR /root
#
COPY ./ /root/UniPSP
#
WORKDIR /root/UniPSP
#
RUN conda env create -f py3.7pytorch1.8.yaml
#SHELL ["conda", "run", "-n", "py3.7pytorch1.8", "/bin/bash", "-c"]
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m nltk.downloader punkt stopwords
#
#WORKDIR /root/UniPSP/third_party/table_pretraining/
#RUN pip install --editable ./

#WORKDIR /
#EXPOSE 2222
#EXPOSE 6000
#EXPOSE 8088
#ENV LANG=en_US.UTF-8
#RUN apt update && \
#    apt install -y \
#        ca-certificates supervisor openssh-server bash ssh \
#        curl wget vim procps htop locales nano man net-tools iputils-ping && \
#    sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen && \
#    locale-gen && \
#    useradd -m -u 13011 -s /bin/bash toolkit && \
#    passwd -d toolkit && \
#    useradd -m -u 13011 -s /bin/bash --non-unique console && \
#    passwd -d console && \
#    useradd -m -u 13011 -s /bin/bash --non-unique _toolchain && \
#    passwd -d _toolchain && \
#    useradd -m -u 13011 -s /bin/bash --non-unique coder && \
#    passwd -d coder && \
#    chown -R toolkit:toolkit /run /etc/shadow /etc/profile && \
#    apt autoremove --purge && apt-get clean && \
#    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
#    echo ssh >> /etc/securetty && \
#    rm -f /etc/legal /etc/motd
#COPY --chown=13011:13011 --from=registry.console.elementai.com/shared.image/sshd:base /tk /tk
#RUN chmod 0600 /tk/etc/ssh/ssh_host_rsa_key
#ENTRYPOINT ["/tk/bin/start.sh"]
# ENTRYPOINT ["bash", "bash_entry.sh"]