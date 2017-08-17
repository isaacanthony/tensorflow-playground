# Construct image. Start instance.
# make start
start:
	docker create -it -p 6006:6006 -v ~/code/tensorflow-playground:/src --name tensorflow-playground tensorflow/tensorflow
	docker start tensorflow-playground

# Run file.
# make run file=test.py
run:
	docker exec -it tensorflow-playground rm -rf /src/logs
	docker exec -it tensorflow-playground python /src/$(file)

# Visualize model using tensorboard on localhost:6006.
# make tensorboard
tensorboard:
	docker exec -it tensorflow-playground tensorboard --logdir /src/logs

# Stop instance. Remove instance.
# make stop
stop:
	docker stop tensorflow-playground
	docker rm tensorflow-playground
