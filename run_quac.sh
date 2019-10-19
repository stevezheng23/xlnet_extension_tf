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
      --numturn=*)
      NUMTURN="${i#*=}"
      shift
      ;;
      --seqlen=*)
      SEQLEN="${i#*=}"
      shift
      ;;
      --querylen=*)
      QUERYLEN="${i#*=}"
      shift
      ;;
      --answerlen=*)
      ANSWERLEN="${i#*=}"
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
      --answerthreshold=*)
      ANSWERTHRESHOLD="${i#*=}"
      shift
      ;;
    esac
  done

echo "gpu device        = ${GPUDEVICE}"
echo "num gpus          = ${NUMGPUS}"
echo "task name         = ${TASKNAME}"
echo "random seed       = ${RANDOMSEED}"
echo "predict tag       = ${PREDICTTAG}"
echo "model dir         = ${MODELDIR}"
echo "data dir          = ${DATADIR}"
echo "output dir        = ${OUTPUTDIR}"
echo "num turn          = ${NUMTURN}"
echo "seq len           = ${SEQLEN}"
echo "query len         = ${QUERYLEN}"
echo "answer len        = ${ANSWERLEN}"
echo "batch size        = ${BATCHSIZE}"
echo "learning rate     = ${LEARNINGRATE}"
echo "train steps       = ${TRAINSTEPS}"
echo "warmup steps      = ${WARMUPSTEPS}"
echo "save steps        = ${SAVESTEPS}"
echo "answer threshold  = ${ANSWERTHRESHOLD}"

alias python=python3
mkdir ${OUTPUTDIR}

start_time=`date +%s`

CUDA_VISIBLE_DEVICES=${GPUDEVICE} python run_quac.py \
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
--num_turn=${NUMTURN} \
--max_seq_length=${SEQLEN} \
--max_query_length=${QUERYLEN} \
--max_answer_length=${ANSWERLEN} \
--train_batch_size=${BATCHSIZE} \
--predict_batch_size=${BATCHSIZE} \
--num_hosts=1 \
--num_core_per_host=${NUMGPUS} \
--learning_rate=${LEARNINGRATE} \
--train_steps=${TRAINSTEPS} \
--warmup_steps=${WARMUPSTEPS} \
--save_steps=${SAVESTEPS} \
--do_train=true \
--do_predict=false \
--do_export=false \
--overwrite_data=false

CUDA_VISIBLE_DEVICES=${GPUDEVICE} python run_quac.py \
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
--num_turn=${NUMTURN} \
--max_seq_length=${SEQLEN} \
--max_query_length=${QUERYLEN} \
--max_answer_length=${ANSWERLEN} \
--train_batch_size=${BATCHSIZE} \
--predict_batch_size=${BATCHSIZE} \
--num_hosts=1 \
--num_core_per_host=1 \
--learning_rate=${LEARNINGRATE} \
--train_steps=${TRAINSTEPS} \
--warmup_steps=${WARMUPSTEPS} \
--save_steps=${SAVESTEPS} \
--do_train=false \
--do_predict=true \
--do_export=false \
--overwrite_data=false

python tool/convert_quac.py \
--input_file=${OUTPUTDIR}/data/predict.${PREDICTTAG}.summary.json \
--output_file=${OUTPUTDIR}/data/predict.${PREDICTTAG}.span.json \
--answer_threshold=${ANSWERTHRESHOLD}

rm ${OUTPUTDIR}/data/predict.${PREDICTTAG}.eval.json

python tool/eval_quac.py \
--val_file=${DATADIR}/dev-${TASKNAME}.json \
--model_output=${OUTPUTDIR}/data/predict.${PREDICTTAG}.span.json \
--o ${OUTPUTDIR}/data/predict.${PREDICTTAG}.eval.json

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

read -n 1 -s -r -p "Press any key to continue..."