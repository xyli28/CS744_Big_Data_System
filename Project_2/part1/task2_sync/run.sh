#!/bin/bash

# cluster_utils.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source cluster_utils.sh
#start_cluster part1_task1.py single
start_cluster part1_task2_sync.py cluster2

# defined in cluster_utils.sh to terminate the cluster
# terminate_cluster
