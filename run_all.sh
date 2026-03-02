python src/main.py task=batch_integration task.foundation_models=[cancerfoundation,cancerfoundation_2,scfoundation,scgpt_human,scgpt_pancancer] task.dataset_name=kim_lung
python src/main.py task=batch_integration task.foundation_models=[cancerfoundation,cancerfoundation_2,scfoundation,scgpt_human,scgpt_pancancer] task.dataset_name=ji_skin
python src/main.py task=batch_integration task.foundation_models=[cancerfoundation,cancerfoundation_2,scfoundation,scgpt_human,scgpt_pancancer] task.dataset_name=neftel_ss2
python src/main.py task=finetuned_batch_integration task.foundation_models=[cancerfoundation,cancerfoundation_2,scgpt_human,scgpt_pancancer] task.dataset_name=kim_lung
python src/main.py task=finetuned_batch_integration task.foundation_models=[cancerfoundation,cancerfoundation_2,scgpt_human,scgpt_pancancer] task.dataset_name=neftel_ss2
python src/main.py task=finetuned_batch_integration task.foundation_models=[cancerfoundation,cancerfoundation_2,scgpt_human,scgpt_pancancer] task.dataset_name=ji_skin
