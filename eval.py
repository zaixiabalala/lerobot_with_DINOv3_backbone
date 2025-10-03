from lerobot.policies.act.modeling_act import ACTPolicy

act_policy = ACTPolicy.from_pretrained("/home/student/workspace/ruogu/lerobot_outputs/train/act_100_2cam_ntem_dumbbell_0930/checkpoints/060000/pretrained_model")

act_policy.eval()
