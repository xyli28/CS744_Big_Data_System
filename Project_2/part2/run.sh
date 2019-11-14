#!/bin/bash

# cluster_utils.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
terminate_cluster
source cluster_utils.sh
start_cluster part2_LeNet.py 2 

# defined in cluster_utils.sh to terminate the cluster
#terminate_cluster

