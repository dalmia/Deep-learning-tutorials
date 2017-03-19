import dtree

tree = dtree.dtree()
party, classes, features = tree.read_data('party.data')
t = tree.make_tree(party, classes, features)
tree.print_tree(t, ' ')

print tree.classify_all(t, party)

for i in range(len(party)):
    tree.classify(t, party[i])

print "True Classes"
print classes
