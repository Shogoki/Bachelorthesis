FROM python:3.7

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq make texlive-full inkscape wget && \
	wget https://github.com/jgm/pandoc/releases/download/2.7.3/pandoc-2.7.3-1-amd64.deb && dpkg -i pandoc-2.7.3-1-amd64.deb && \
	PYTHONIOENCODING=UTF-8 pip3 install numpy pandas matplotlib pyocclient
WORKDIR /usr/src/app
ENV owncloud-password "default-pw"
CMD ["make", "pipeline" ]	

