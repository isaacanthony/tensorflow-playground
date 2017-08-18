# Construct image. Start instance.
# make start
start:
	docker create -it -p 6006:6006 -v ~/code/tensorflow-playground:/src -w /src --name tensorflow-playground tensorflow/tensorflow
	docker start tensorflow-playground

# Run file.
# make run file=test.py
run:
	docker exec -it tensorflow-playground rm -rf logs
	docker exec -it tensorflow-playground python $(file)

# Visualize model using tensorboard on localhost:6006.
# make tensorboard
tensorboard:
	docker exec -it tensorflow-playground tensorboard --logdir logs

# SSH into docker instance.
# make bash
bash:
	docker exec -it tensorflow-playground bash

# Stop instance. Remove instance.
# make stop
stop:
	docker stop tensorflow-playground
	docker rm tensorflow-playground
