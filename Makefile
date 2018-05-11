start:
	@docker create -it -p 6006:6006 -v ~/code/tensorflow-playground/src:/src -w /src --name tensorflow-playground tensorflow/tensorflow:1.8.0-py3
	@docker start tensorflow-playground

run:
	@docker exec -it tensorflow-playground python3 $(file).py $(arg)

tensorboard:
	@docker exec -it tensorflow-playground tensorboard --logdir logs

tfjs:
	$(eval dir=`echo "$(file)" | sed -e "s/\.h5//g"`)
	@docker exec -it tensorflow-playground pip3 install tensorflowjs
	@docker exec -it tensorflow-playground tensorflowjs_converter --input_format keras $(file) $(dir)

server:
	@docker exec -it tensorflow-playground python3 -m http.server 6006

bash:
	@docker exec -it tensorflow-playground bash

stop:
	@docker exec -it tensorflow-playground rm -rf logs
	@docker stop tensorflow-playground
	@docker rm tensorflow-playground
