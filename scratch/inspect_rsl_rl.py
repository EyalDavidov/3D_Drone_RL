import inspect
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

def main():
    dist_cfg = RslRlMLPModelCfg.GaussianDistributionCfg
    print("GaussianDistributionCfg fields and documentation:")
    try:
        source = inspect.getsource(dist_cfg)
        print(source)
    except Exception as e:
        print("Error getting source:", e)
        print("Attributes:", dir(dist_cfg))
        # Print class annotations
        if hasattr(dist_cfg, '__annotations__'):
            print("Annotations:", dist_cfg.__annotations__)

if __name__ == '__main__':
    main()
