start:
	@docker create -it -p 6006:6006 -v ~/code/tensorflow-playground/src:/src -w /src --name tensorflow-playground tensorflow/tensorflow:1.7.0-py3
	@docker start tensorflow-playground

run:
	@docker exec -it tensorflow-playground python3 $(file).py $(arg)

tensorboard:
	@docker exec -it tensorflow-playground tensorboard --logdir logs

bash:
	@docker exec -it tensorflow-playground bash

stop:
	@docker exec -it tensorflow-playground rm -rf logs
	@docker stop tensorflow-playground
	@docker rm tensorflow-playground
