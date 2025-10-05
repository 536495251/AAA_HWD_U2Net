import datetime
import subprocess
import sys
import os
from apscheduler.schedulers.blocking import BlockingScheduler
def run_train_project():
    """运行训练项目的主函数"""
    print(f"开始执行训练脚本 - {datetime.datetime.now()}")

    # 替换为你的train.py实际路径
    train_script_path = "/home/huang/work/InfraredSmallTargets/Code/MyProject/train/train.py"

    # 检查文件是否存在
    if not os.path.exists(train_script_path):
        print(f"错误：找不到文件 {train_script_path}")
        return
    try:
        # 运行train.py
        result = subprocess.run(
            [sys.executable, train_script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(train_script_path)  # 在train.py所在目录运行
        )

        # 输出结果
        if result.stdout:
            print("输出内容:")
            print(result.stdout)

        if result.stderr:
            print("错误信息:")
            print(result.stderr)

        print(f"✅ 训练脚本执行完成 - {datetime.datetime.now()}")

    except Exception as e:
        print(f"❌ 执行过程中发生错误: {e}")


def main(train_script_path=None):
    """主函数：设置定时任务"""
    # 创建调度器
    scheduler = BlockingScheduler()
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    target_time = tomorrow.replace(hour=10, minute=0, second=0, microsecond=0)
    # target_time = now.replace(hour=20, minute=30, second=0, microsecond=0)

    print("定时任务设置信息:")
    print(f"   当前时间: {now}")
    print(f"   目标时间: {target_time}")
    print(f"   将在 {(target_time - now).total_seconds():.0f} 秒后执行")
    print("   Train.py路径:" )
    print(train_script_path)
    print("   CTRL+C 可以取消定时任务")

    # 添加一次性定时任务
    scheduler.add_job(
        run_train_project,
        'date',
        run_date=target_time,
        id='train_job'
    )
    try:
        # 启动调度器
        scheduler.start()
    except KeyboardInterrupt:
        print("\n⏹️ 定时任务已被用户取消")
    except Exception as e:
        print(f"\n❌ 调度器错误: {e}")
if __name__ == "__main__":
    path="/home/huang/work/InfraredSmallTargets/Code/MyProject/train/train.py"
    main(path)