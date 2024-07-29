from matplotlib.pyplot import Axes
from matplotlib.text import Annotation
import re
import numpy as np

def format_decision_tree_plot(axes :Axes) -> None:
    '''
    Formats the decision tree so that each node:
    * only displays the mode cluster in the sample at that node
    * breaks up the decision variable into multiple lines
    * does not display the number of samples at that node.
    Operates in place.
    '''
    for property in axes.properties()['children']:
        # for every axes property that is a `Text` object,
        if type(property) == Annotation:
            
            text_content = property.get_text()

            #--------------------------------------------------------------------#
            # Breaking up long skill names into multiple lines
            #--------------------------------------------------------------------#
            skill_name_position = re.search(pattern="[><]", string=text_content)
            if skill_name_position:
                skill_name_end_index = skill_name_position.start() - 1
                skill_name = text_content[:skill_name_end_index + 1].replace(' ', '\n')
                text_content = skill_name + text_content[skill_name_end_index:]

            #--------------------------------------------------------------------#
            # Delete "samples" detail in each node (e.g. samples = 32)
            #--------------------------------------------------------------------#
            sample_detail = re.search(
                pattern="(samples = )[^(\\n)]*(\\n)",
                string=text_content
            )
            text_content = text_content[0:sample_detail.start()] + text_content[sample_detail.end():]

            #--------------------------------------------------------------------#
            # Replace array of cluster assignments with the mode cluster
            # and rename from "value" to "cluster"
            #--------------------------------------------------------------------#
            # find the index in text where the cluster assignments array is displayed
            values_start_index = re.search(
                pattern="(value = \[)[^\]]*(\])", 
                string=text_content
            ).start()
            # calculate the most popular cluster in this node's subsample
            most_popular_class = np.array(
                re.split(
                    pattern=', |\\n',
                    string = text_content[values_start_index + 8:][1:-1]
                )
            ).astype('float').argmax()
            # in the text, replace the values array with just the most popular cluster
            property.set_text(
                text_content[:values_start_index] + 'cluster = ' + str(most_popular_class)
            )