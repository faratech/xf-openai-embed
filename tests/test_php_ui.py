"""PHP rendering probes replace all cURL functions before including the page."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which('php') is None, reason='PHP CLI is unavailable')
@pytest.mark.parametrize('query,expected_calls', [('0', 1), ([], 0), ('', 0)])
def test_ui_accepts_zero_and_rejects_array_input_without_backend_access(query, expected_calls):
    page = str(Path(__file__).resolve().parents[1] / 'ai-search.php')
    code = '''
    $calls = 0;
    function curl_init($url) { global $calls; ++$calls; return true; }
    function curl_setopt(...$args) {}
    function curl_exec($ch) { return '{}'; }
    function curl_getinfo(...$args) { return 200; }
    function curl_close(...$args) {}
    $_SERVER['REQUEST_METHOD'] = 'POST';
    $_POST = json_decode($argv[1], true);
    ob_start();
    include $argv[2];
    ob_end_clean();
    echo $calls;
    '''
    result = subprocess.run(['php', '-n', '-d',
                             'disable_functions=curl_init,curl_setopt,curl_exec,curl_getinfo,curl_close',
                             '-r', code, json.dumps({'query': query, 'mode': []}), page],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout == str(expected_calls)
