# tensorflow-playground
Dumping ground for tensorflow test scripts

## Commands
* `make start` Creates and starts a docker instance called tensorflow-playground.
* `make run file=hello_world` Runs hello_world.py in the docker instance.
* `make tensorboard` Starts tensorboard on localhost:6006.
* `make tfjs file=model.h5` Converts model.h5 to model/model.json.
* `make server` Starts HTTP server on localhost:6006.
* `make bash` Opens a bash terminal in the docker instance.
* `make stop` Stops and removes the docker instance.
