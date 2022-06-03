
# python -m src_task1.test_exp3 \
#     --dataroot /home/suparna/work/dataset \
#     --exp_name exp3_run2 \
#     --batch_size 20 \
#     --ckpt best \
#     --num_classes 19




###### task 2 #######
python -m src_task2.test_exp1 \
    --dataroot /home/suparna/work/dataset \
    --exp_name exp1_task2_run1 \
    --batch_size 20 \
    --ckpt best \
    --num_classes 19 \
    --issum True
