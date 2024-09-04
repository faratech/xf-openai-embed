<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            transition: background-color 0.3s, color 0.3s;
        }
        body.light {
            background-color: #f4f4f4;
            color: #333;
        }
        body.dark {
            background-color: #222;
            color: #ddd;
        }
        form {
            margin-bottom: 20px;
        }
        input[type="text"], input[type="submit"] {
            padding: 8px;
            font-size: 16px;
        }
        input[type="text"] {
            width: 300px;
        }
        input[type="submit"] {
            cursor: pointer;
        }
        .theme-switcher {
            margin-bottom: 20px;
        }
        .split-view {
            display: flex;
            gap: 20px;
        }
        .normal-output, .json-output {
            flex: 1;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 8px;
            max-height: 400px;
            overflow-y: auto;
        }
        .json-output {
            background-color: #282c34;
            color: #61dafb;
        }
        pre {
            white-space: pre-wrap; /* Allows wrapping */
            word-wrap: break-word; /* Break long words */
        }
        h1, h2, h3 {
            margin-top: 0;
        }
    </style>
</head>
<body class="light">
    <h1>Search</h1>

    <div class="theme-switcher">
        <label for="theme">Choose Theme:</label>
        <select id="theme" onchange="toggleTheme()">
            <option value="light">Light</option>
            <option value="dark">Dark</option>
        </select>
    </div>

    <form method="POST" action="">
        <input type="text" name="query" placeholder="Enter search term" required>
        <input type="submit" value="Search">
    </form>

    <div class="split-view">
        <div class="normal-output">
            <h2>Search Results</h2>
            <?php
            if ($_SERVER['REQUEST_METHOD'] === 'POST' && !empty($_POST['query'])) {
                $query = $_POST['query'];
                $max_results = 10;

                $url = 'http://127.0.0.1:8000/faiss/search/';

                $data = array(
                    'query' => $query,
                    'max_results' => $max_results
                );

                $json_data = json_encode($data);

                $ch = curl_init($url);
                curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
                curl_setopt($ch, CURLOPT_POST, true);
                curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
                curl_setopt($ch, CURLOPT_POSTFIELDS, $json_data);

                $response = curl_exec($ch);
                $http_status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
                curl_close($ch);

                if ($http_status == 200) {
                    $results = json_decode($response, true);
                    
                    if (!empty($results['faiss_results'])) {
                        foreach ($results['faiss_results'] as $result) {
                            echo '<div class="result">';
                            echo '<h3>' . htmlspecialchars($result['thread_title']) . '</h3>';
                            echo '<p>' . htmlspecialchars($result['message']) . '</p>';
                            echo '<p class="meta">Post Date: ' . date('Y-m-d H:i:s', $result['post_date']) . '</p>';
                            echo '</div>';
                        }
                    } else {
                        echo '<p>No results found for "' . htmlspecialchars($query) . '".</p>';
                    }
                } else {
                    echo '<p>An error occurred while processing the search. Please try again later.</p>';
                }
            }
            ?>
        </div>

        <div class="json-output">
            <h2>JSON Output</h2>
            <pre><?php
            if (!empty($response)) {
                // Pretty print the JSON response
                echo htmlspecialchars(json_encode(json_decode($response), JSON_PRETTY_PRINT));
            }
            ?></pre>
        </div>
    </div>

    <script>
        function toggleTheme() {
            const theme = document.getElementById('theme').value;
            document.body.className = theme;
        }
    </script>
</body>
</html>
