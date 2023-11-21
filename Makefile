.DEFAULT_GOAL = run

build:
		docker build . -t optimistic_init

run:
		docker run --rm -itd optimistic_init
