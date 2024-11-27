Based on https://huggingface.co/spaces/akazakov/rag-gradio-sample-project/tree/main/gradio_app

## About
...

## Deploy with docker compose

### Prerequisites

Make

[Docker](https://docs.docker.com/engine/install/ubuntu/)

[Docker compose](https://docs.docker.com/compose/install/)

### Environment Variables
To run this project, you will need to add the following environment variables to your .env file.


`OPENAI_API_ENDPOINT_URL`

Example .env file

```bash
OPENAI_API_ENDPOINT_URL=http://172.17.0.1:8080/v1/completions
```

## Deployment (docker compose)

To deploy run

```bash
make deploy
```

To delete deployment run
```bash
make undeploy
```

To stop deployment run
```bash
make stop
```
