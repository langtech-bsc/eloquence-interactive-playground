python -m prep_scripts.prepare_users

export GRADIO_SERVER_PORT=8080; python -m gradio_app.app &
sleep 5
export GRADIO_SERVER_PORT=8081; python -m gradio_app.app_upload
