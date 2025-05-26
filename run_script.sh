python -m prep_scripts.prepare_users

python -m dialogue_manager.dummy &
sleep 5
python -m gradio_app.backend.ws_audio_server &
sleep 5
python -m retrievers.retrieval_server &
sleep 5
export GRADIO_SERVER_PORT=8080; python -m gradio_app.app 
