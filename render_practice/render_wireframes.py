from nicegui import ui


struct = {
  "type": "page",
  "children": [
    {"type": "header"},
    {"type": "section", "children": [
        {"type": "textfield"},
        {"type": "button"}
    ]},
    {"type": "footer"}
  ]
}

def render(structure):
    if structure['type']=='page':
        with ui.card():
            ui.label('Title')
            for node in structure.get('children',[]):
                render(node)
    elif structure['type']=='section':
        with ui.card():
            for node in structure.get('children',[]):
                render(node)
    elif structure['type']=='text':
        ui.label('This is text')
    elif structure['type']=='button':
        ui.button('login')
    elif structure['type']=='textfield':
        ui.input()

    
render(struct)


ui.run()