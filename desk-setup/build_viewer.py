"""Inject desk_spec.json into the viewer template -> four-mode-desk.html"""
import os
here = os.path.dirname(os.path.abspath(__file__))
spec = open(os.path.join(here, 'desk_spec.json'), encoding='utf-8').read().replace('</', '<' + '\\' + '/')
tpl = open(os.path.join(here, 'viewer_template.html'), encoding='utf-8').read()
out = tpl.replace('/*SPEC*/', spec)
open(os.path.join(here, 'four-mode-desk.html'), 'w', encoding='utf-8').write(out)
print('wrote four-mode-desk.html', len(out))
