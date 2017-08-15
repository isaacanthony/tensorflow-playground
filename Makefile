# Construct image. Start instance.
# make start
start:
	docker create -it -v ~/code/tensorflow-playground:/src --name tensorflow-playground tensorflow/tensorflow
	docker start tensorflow-playground

# Run file.
# make run file=test.py
run:
	docker exec -it tensorflow-playground python /src/$(file)

# Stop instance. Remove instance.
# make stop
stop:
	docker stop tensorflow-playground
	docker rm tensorflow-playground
