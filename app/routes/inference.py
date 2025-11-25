from app.routes import router


@router.post("/infer_instances")
async def infer_instances(seed_instances: list[list[int]]):
    """ Infer instances from seed instances. """
    pass
