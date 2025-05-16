import asyncio

from autogenstudio.teammanager import TeamManager

async def run_team():
    # Initialize the TeamManager
    manager = TeamManager()

    # Run a task with a specific team configuration
    result = await manager.run(
    task="Tell me about the trinity in Catholocism?",
    team_config="team.json"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(run_team())
