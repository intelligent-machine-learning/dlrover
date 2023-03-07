kubectl delete -f scale_plan.yaml -n dlrover
kubectl delete -f deepctr_auto_scale_job.yaml -n dlrover
kubectl delete  po ` kubectl get po -n dlrover | grep deepctr-sample-auto-scaling | awk '{print $1}' ` -n dlrover  --force
sleep 100
cd /Users/hanxudong/code/iml/dockerfile/ack_test_docker
sh run.sh
cd /Users/hanxudong/code/iml/dlrover/dlrover/examples
kubectl apply -f deepctr_auto_scale_job.yaml -n dlrover
sleep 100
kubectl apply -f scale_plan.yaml -n dlrover
kubectl get pod -n dlrover 
kubectl logs -n dlrover deepctr-auto-scaling-job-edljob-chief-0 > chief_log
kubectl logs -n dlrover elasticjob-deepctr-auto-scaling-job-dlrover-master > log_master