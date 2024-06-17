#!/bin/bash

FILE=$1


function get_hyperparam() {
    grep ".* \* $1: " $FILE | sed "s/.*: //"
}


grep average $1 | sed 's/, lr:.*//;s/.*: //' | \
    paste <(get_hyperparam "weight_decay") <(get_hyperparam "lr_warmup_step_ratio") <(get_hyperparam "lr") <(get_hyperparam "gradient_accumulation_steps") <(get_hyperparam "max_grad_norm") -


# Weight decay
# Warmup steps
# Learning rate
# Gradent accumulation steps
