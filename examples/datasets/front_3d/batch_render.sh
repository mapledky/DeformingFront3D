args_file="code/BlenderProc/examples/datasets/front_3d/config/deformingfront.yaml"
# 检查是否传递了参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_executions>"
    exit 1
fi

# 获取输入参数
NUM_EXECUTIONS=$1


# 获取当前时间戳并创建新的输出文件夹
# timestamp=$(date +"%m-%d-%H-%M")
# output_dir="${output_dir}/${timestamp}"
# mkdir -p "$output_dir"

# 多次执行Python文件
# for ((i=0; i<NUM_EXECUTIONS; i++)); do
#     echo "Executing Python script: run $((i+1))"
#     blenderproc run code/BlenderProc/examples/datasets/front_3d/main_pipe.py "$front_folder" "$future_folder" "$output_dir" "$anime_folder" "$shapenet_folder" "$shapenet_json" "$dt4_json" "$anime_folder_animals"
#     echo "convert depth to point"
    
# done
for ((i=0; i<100; i++)); do
    echo "Executing Python script: run $((i+1))"
    blenderproc run code/BlenderProc/examples/datasets/front_3d/main_pipe.py "$args_file"
    echo "convert depth to point"
    
done

