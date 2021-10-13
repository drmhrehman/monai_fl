#FROM pwrightkcl/test:preprocess
FROM pwrightkcl/spm-clinical:20210625
MAINTAINER Mark Graham "mark.graham@kcl.ac.uk"

# Create a non-root user
ARG username=docker
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN \
    adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

# Set user
USER $USER


COPY wrapper_script.sh /home
#ENTRYPOINT ["/home/wrapper_script.sh"]

ENTRYPOINT ["/bin/bash"]
