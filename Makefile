deploy:
	docker compose --env-file ./pilot3/.env up -d --build
undeploy:
	docker compose down
stop:
	docker compose stop