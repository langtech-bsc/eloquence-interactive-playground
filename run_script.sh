python -m prep_scripts.prepare_users

python -m dialogue_manager.dummy &
sleep 5
python -m retrievers.retrieval_server &
python -m retrievers.retrieval_server --endpoint "http://localhost:7998" &
sleep 5
export GRADIO_SERVER_PORT=8080; python -m gradio_app.app 
