class ListService():

    def recursion_toList(self, obj, attr):
        current = obj
        yield current
        while True:
            current = getattr(current, attr, None)
            if current is not None:
                yield current
            else:
                return