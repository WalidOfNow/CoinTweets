import grequests
import random

urls=[
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000',
]

text_vals = []

def action(resp, timeout, verify, proxies, stream, cert):
    text_vals.append (resp.text)

def exception_handler(request, exception):
    print (exception)


async_list = []

for u in urls:
    # The "hooks = {..." part is where you define what you want to do
    #
    # Note the lack of parentheses following do_something, this is
    # because the response will be used as the first argument automatically
    action_item = grequests.get(u, hooks = {'response' : action})

    # Add the task to our list of things to do via async
    async_list.append(action_item)


# Do our list of things to do via async
grequests.map(async_list, exception_handler=exception_handler)

# verify all different outputs
if (len(set(text_vals)) != len(text_vals)):
    print("duplicated text, async issue possibly detected")
else:
    print("no async issues found, sent {0} successful requsts".format(str(len(text_vals))))