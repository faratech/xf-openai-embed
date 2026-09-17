<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XenForo AI Search (Semantic & Hybrid)</title>
    <style>
        :root {
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #212529;
            --border-color: #dee2e6;
            --primary: #0d6efd;
            --badge-bg: #e9ecef;
            --badge-color: #495057;
            --meta-color: #6c757d;
        }
        body.dark {
            --bg-color: #121212;
            --card-bg: #1e1e1e;
            --text-color: #e0e0e0;
            --border-color: #333333;
            --primary: #3d8bfd;
            --badge-bg: #2d3135;
            --badge-color: #a0aec0;
            --meta-color: #888888;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 24px;
            transition: background-color 0.2s, color 0.2s;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }
        h1 { margin: 0; font-size: 24px; font-weight: 600; }
        .theme-select {
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            background: var(--card-bg);
            color: var(--text-color);
        }
        .search-form {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            flex: 1;
            min-width: 280px;
            padding: 10px 14px;
            font-size: 16px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: var(--card-bg);
            color: var(--text-color);
        }
        .search-form select, .search-form input[type="submit"] {
            padding: 10px 16px;
            font-size: 15px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: var(--card-bg);
            color: var(--text-color);
        }
        .search-form input[type="submit"] {
            background-color: var(--primary);
            color: white;
            border: none;
            cursor: pointer;
            font-weight: 500;
        }
        .split-view {
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 24px;
        }
        @media (max-width: 900px) {
            .split-view { grid-template-columns: 1fr; }
        }
        .results-panel, .json-panel {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .results-panel {
            max-height: 700px;
            overflow-y: auto;
        }
        .result-card {
            border-bottom: 1px solid var(--border-color);
            padding: 14px 0;
        }
        .result-card:last-child { border-bottom: none; }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 8px;
        }
        .result-header h3 {
            margin: 0 0 6px 0;
            font-size: 17px;
            color: var(--primary);
        }
        .score-badge {
            font-size: 12px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 12px;
            background: var(--badge-bg);
            color: var(--badge-color);
            white-space: nowrap;
        }
        .result-message {
            margin: 6px 0;
            font-size: 14px;
            line-height: 1.5;
        }
        .result-meta {
            font-size: 12px;
            color: var(--meta-color);
        }
        .json-panel pre {
            margin: 0;
            padding: 12px;
            background-color: #1e1e1e;
            color: #4ec9b0;
            border-radius: 6px;
            max-height: 640px;
            overflow-y: auto;
            font-size: 13px;
            font-family: "Courier New", Courier, monospace;
        }
    </style>
</head>
<body class="light">
    <div class="container">
        <header>
            <h1>XenForo Semantic & Hybrid Search</h1>
            <div>
                <label for="theme">Theme: </label>
                <select id="theme" class="theme-select" onchange="toggleTheme()">
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                </select>
            </div>
        </header>

        <?php
        $query = isset($_POST['query']) ? trim($_POST['query']) : '';
        $mode = isset($_POST['mode']) ? $_POST['mode'] : 'combined';
        $max_results = 10;
        $response = null;
        $results = [];
        $http_status = 0;

        if ($_SERVER['REQUEST_METHOD'] === 'POST' && !empty($query)) {
            $endpoint = ($mode === 'elastic') ? 'elastic/' : (($mode === 'vector') ? 'search/' : 'combined/');
            $url = 'http://127.0.0.1:8000/faiss/' . $endpoint;

            $data = [
                'query' => $query,
                'max_results' => $max_results
            ];

            $ch = curl_init($url);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
            curl_setopt($ch, CURLOPT_POST, true);
            curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
            curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
            curl_setopt($ch, CURLOPT_TIMEOUT, 10);

            $response = curl_exec($ch);
            $http_status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
            curl_close($ch);

            if ($http_status === 200 && $response) {
                $decoded = json_decode($response, true);
                if (isset($decoded['combined_results'])) {
                    $results = $decoded['combined_results'];
                } elseif (isset($decoded['faiss_results'])) {
                    $results = $decoded['faiss_results'];
                } elseif (isset($decoded['elasticsearch_results'])) {
                    $results = $decoded['elasticsearch_results'];
                }
            }
        }
        ?>

        <form class="search-form" method="POST" action="">
            <input type="text" name="query" placeholder="Enter search query..." value="<?php echo htmlspecialchars($query); ?>" required>
            <select name="mode">
                <option value="combined" <?php if ($mode === 'combined') echo 'selected'; ?>>Hybrid (RRF)</option>
                <option value="vector" <?php if ($mode === 'vector') echo 'selected'; ?>>Semantic Vector</option>
                <option value="elastic" <?php if ($mode === 'elastic') echo 'selected'; ?>>Elasticsearch BM25</option>
            </select>
            <input type="submit" value="Search">
        </form>

        <div class="split-view">
            <div class="results-panel">
                <h2>Search Results <?php if (!empty($results)) echo '(' . count($results) . ')'; ?></h2>
                <?php
                if ($_SERVER['REQUEST_METHOD'] === 'POST') {
                    if ($http_status === 200) {
                        if (!empty($results)) {
                            foreach ($results as $r) {
                                $title = !empty($r['thread_title']) ? $r['thread_title'] : 'Post #' . ($r['post_id'] ?? 'N/A');
                                $message = !empty($r['message']) ? $r['message'] : '';
                                $date_str = !empty($r['post_date']) ? date('Y-m-d H:i:s', $r['post_date']) : 'Unknown';
                                $score = isset($r['score']) ? number_format($r['score'], 4) : '';
                                $type = strtoupper($r['type'] ?? 'POST');

                                echo '<div class="result-card">';
                                echo '  <div class="result-header">';
                                echo '    <h3>' . htmlspecialchars($title) . '</h3>';
                                if ($score !== '') {
                                    echo '    <span class="score-badge">Score: ' . htmlspecialchars($score) . '</span>';
                                }
                                echo '  </div>';
                                if ($message) {
                                    echo '  <div class="result-message">' . htmlspecialchars(mb_strimwidth($message, 0, 240, "...")) . '</div>';
                                }
                                echo '  <div class="result-meta">';
                                echo '    <span>[' . htmlspecialchars($type) . ']</span> ';
                                if (!empty($r['post_id'])) echo '<span>Post ID: ' . intval($r['post_id']) . '</span> &bull; ';
                                if (!empty($r['thread_id'])) echo '<span>Thread ID: ' . intval($r['thread_id']) . '</span> &bull; ';
                                echo '    <span>Date: ' . htmlspecialchars($date_str) . '</span>';
                                echo '  </div>';
                                echo '</div>';
                            }
                        } else {
                            echo '<p>No results found for "' . htmlspecialchars($query) . '".</p>';
                        }
                    } else {
                        echo '<p style="color:#dc3545;">Failed to reach search service (HTTP ' . intval($http_status) . '). Ensure the FastAPI service is running on port 8000.</p>';
                    }
                } else {
                    echo '<p style="color:var(--meta-color);">Enter a query above to search using modern semantic and hybrid ranking.</p>';
                }
                ?>
            </div>

            <div class="json-panel">
                <h2>Raw JSON Response</h2>
                <pre><?php
                if (!empty($response)) {
                    echo htmlspecialchars(json_encode(json_decode($response), JSON_PRETTY_PRINT));
                } else {
                    echo "// JSON payload will appear here after search";
                }
                ?></pre>
            </div>
        </div>
    </div>

    <script>
        function toggleTheme() {
            const theme = document.getElementById('theme').value;
            document.body.className = theme;
            localStorage.setItem('xf_search_theme', theme);
        }
        const savedTheme = localStorage.getItem('xf_search_theme');
        if (savedTheme) {
            document.getElementById('theme').value = savedTheme;
            document.body.className = savedTheme;
        }
    </script>
</body>
</html>
