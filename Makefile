deploy:
	docker compose --env-file .env up -d --build
undeploy:
	docker compose down
stop:
	docker compose stop