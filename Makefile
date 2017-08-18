start:
	docker create -it -p 6006:6006 -v ~/code/tensorflow-playground:/src -w /src --name tensorflow-playground tensorflow/tensorflow:1.3.0
	docker start tensorflow-playground

run:
	docker exec -it tensorflow-playground rm -rf logs
	docker exec -it tensorflow-playground python $(file)

tensorboard:
	docker exec -it tensorflow-playground tensorboard --logdir logs

bash:
	docker exec -it tensorflow-playground bash

stop:
	docker stop tensorflow-playground
	docker rm tensorflow-playground
