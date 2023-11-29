#!/bin/bash

docker run -itd --rm \
	-v "$(pwd)":/workspace \
	optimistic_init

