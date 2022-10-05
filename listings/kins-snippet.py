from psyki.ski import Injector
from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string

# ...

# For this algorithm we need to explicitly specify the mapping
# between feature names and variable names
feature_mapping = {...}

# Symbolic knowledge
with open(filename) as f:
    rows = f.readlines()
# 1 - Parse textual logic rules into visitable Formulae
knowledge = [get_formula_from_string(row) for row in rows]

predictor = create_fully_connected_nn()
# 2 and 3 - Injector creation (internal fuzzification) and injection
injector = Injector.kins(predictor, feature_mapping)
predictor_with_knowledge = injector.inject(knowledge)

# 4 - Training
predictor_with_knowledge.fit(train_x, train_y)
