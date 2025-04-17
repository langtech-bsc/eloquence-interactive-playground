docker build -t eloquence-interactive-playground-gradio/add-llm .
docker run -d  --net="host" --mount type=volume,src=eloquence-playground,dst=/app/eloquence-playground eloquence-interactive-playground-gradio/add-llm
