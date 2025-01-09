
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    kill -9 $pid
done
