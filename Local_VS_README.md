# Preparation
The running container needs a mounted volume to be able to store the indexes permanently.
First, let's create a dedicated directory for this purpose.
```
mkdir persistent_data_root
```

# Getting the image
## Pulling from Docker hub
The latest version of the image can be pulled form docker hub using command:
```
docker pull vhudecekomilia/eloquence-local-vs:latest
```
This prepared image has port exposed on 8030.


## Build from Dockerfile
You can also build the image yourself using a standard Dockerfile, specifying what port will be exposed to communicate with the outside world.
```
docker build -t eloquence-local-vs --build-arg EXPOSED_PORT=8030 -f Dockerfile-local-vs .
```

# Running the container
Once we have the image and persitent directory, we can run the container
```
docker run -d -v persistent_data_root:/data eloquence-local-vs
```
Based on your networking setting, you might also need to add `--net=host` in order to be able to connect to the container.
Now, the container is running a webserver which is bind to "http://localhost:${EXPOSED_PORT}".
You should make sure that the incoming internet traffic is forwarded to this port.