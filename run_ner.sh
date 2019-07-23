for i in "$@"
  do
    case $i in
      -g=*|--gpudevice=*)
      GPUDEVICE="${i#*=}"
      shift
      ;;
      -n=*|--numgpus=*)
      NUMGPUS="${i#*=}"
      shift
      ;;
      -t=*|--taskname=*)
      TASKNAME="${i#*=}"
      shift
      ;;
      -r=*|--randomseed=*)
      RANDOMSEED="${i#*=}"
      shift
      ;;
      -p=*|--predicttag=*)
      PREDICTTAG="${i#*=}"
      shift
      ;;
      -m=*|--modeldir=*)
      MODELDIR="${i#*=}"
      shift
      ;;
      -d=*|--datadir=*)
      DATADIR="${i#*=}"
      shift
      ;;
      -o=*|--outputdir=*)
      OUTPUTDIR="${i#*=}"
      shift
      ;;
      --maxlen=*)
      MAXLEN="${i#*=}"
      shift
      ;;
      --batchsize=*)
      BATCHSIZE="${i#*=}"
      shift
      ;;
      --learningrate=*)
      LEARNINGRATE="${i#*=}"
      shift
      ;;
      --trainsteps=*)
      TRAINSTEPS="${i#*=}"
      shift
      ;;
      --warmupsteps=*)
      WARMUPSTEPS="${i#*=}"
      shift
      ;;
      --savesteps=*)
      SAVESTEPS="${i#*=}"
      shift
      ;;
    esac
  done

echo "gpu device     = ${GPUDEVICE}"
echo "num gpus       = ${NUMGPUS}"
echo "task name      = ${TASKNAME}"
echo "random seed    = ${RANDOMSEED}"
echo "predict tag    = ${PREDICTTAG}"
echo "model dir      = ${MODELDIR}"
echo "data dir       = ${DATADIR}"
echo "output dir     = ${OUTPUTDIR}"
echo "max len        = ${MAXLEN}"
echo "batch size     = ${BATCHSIZE}"
echo "learning rate  = ${LEARNINGRATE}"
echo "train steps    = ${TRAINSTEPS}"
echo "warmup steps   = ${WARMUPSTEPS}"
echo "save steps     = ${SAVESTEPS}"

alias python=python3

start_time=`date +%s`

CUDA_VISIBLE_DEVICES=${GPUDEVICE} python run_ner.py \
--spiece_model_file=${MODELDIR}/spiece.model \
--model_config_path=${MODELDIR}/xlnet_config.json \
--init_checkpoint=${MODELDIR}/xlnet_model.ckpt \
--task_name=${TASKNAME} \
--random_seed=${RANDOMSEED} \
--predict_tag=${PREDICTTAG} \
--lower_case=false \
--data_dir=${DATADIR}/ \
--output_dir=${OUTPUTDIR}/data \
--model_dir=${OUTPUTDIR}/checkpoint \
--export_dir=${OUTPUTDIR}/export \
--max_seq_length=${MAXLEN} \
--train_batch_size=${BATCHSIZE} \
--eval_batch_size=${BATCHSIZE} \
--predict_batch_size=${BATCHSIZE} \
--num_hosts=1 \
--num_core_per_host=${NUMGPUS} \
--learning_rate=${LEARNINGRATE} \
--train_steps=${TRAINSTEPS} \
--warmup_steps=${WARMUPSTEPS} \
--save_steps=${SAVESTEPS} \
--do_train=true \
--do_eval=false \
--do_predict=false \
--do_export=false \
--overwrite_data=false

CUDA_VISIBLE_DEVICES=${GPUDEVICE} python run_ner.py \
--spiece_model_file=${MODELDIR}/spiece.model \
--model_config_path=${MODELDIR}/xlnet_config.json \
--init_checkpoint=${MODELDIR}/xlnet_model.ckpt \
--task_name=${TASKNAME} \
--random_seed=${RANDOMSEED} \
--predict_tag=${PREDICTTAG} \
--lower_case=false \
--data_dir=${DATADIR}/ \
--output_dir=${OUTPUTDIR}/data \
--model_dir=${OUTPUTDIR}/checkpoint \
--export_dir=${OUTPUTDIR}/export \
--max_seq_length=${MAXLEN} \
--train_batch_size=${BATCHSIZE} \
--eval_batch_size=${BATCHSIZE} \
--predict_batch_size=${BATCHSIZE} \
--num_hosts=1 \
--num_core_per_host=1 \
--learning_rate=${LEARNINGRATE} \
--train_steps=${TRAINSTEPS} \
--warmup_steps=${WARMUPSTEPS} \
--save_steps=${SAVESTEPS} \
--do_train=false \
--do_eval=true \
--do_predict=true \
--do_export=false \
--overwrite_data=false

python tool/convert_token.py \
--input_file=${OUTPUTDIR}/data/predict.${PREDICTTAG}.json \
--output_file=${OUTPUTDIR}/data/predict.${PREDICTTAG}.txt

python tool/eval_token.py \
< ${OUTPUTDIR}/data/predict.${PREDICTTAG}.txt \
> ${OUTPUTDIR}/data/predict.${PREDICTTAG}.token

read -n 1 -s -r -p "Press any key to continue..."