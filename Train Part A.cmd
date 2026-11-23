@echo off
rem Trains every Part A algorithm in parallel: one window per algorithm, seeds run in sequence.
rem Each process uses 2 CPU threads. Finished runs are skipped, so you can close a window and rerun later.
rem To train only your own algorithms instead:  python -m medinav train --config configs\part_a.yaml --only ppo a2c
cd /d "%~dp0"
if not exist logs mkdir logs
for %%A in (q_learning sarsa dqn double_dqn dueling_dqn ppo a2c sac td3) do (
  start "train %%A" cmd /c ".venv\Scripts\python -m medinav train --config configs\part_a.yaml --only %%A > logs\part_a_%%A.log 2>&1"
)
echo Started 9 training windows. Logs: logs\part_a_*.log
echo When they have all closed, run:  python -m medinav pipeline --config configs\part_a.yaml
pause
