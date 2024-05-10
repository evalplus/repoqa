package main


import (
   "encoding/json"
   "go/ast"
   "go/parser"
   "go/token"
   "io/fs"
   "os"
   "path/filepath"
   "strings"
)


type DependencyMap struct {
   Name     string              `json:"name"`
   RepoName map[string][]string `json:"repoName"`
}


func main() {
   if len(os.Args) != 3 {
       os.Stderr.WriteString("Usage: go run main.go <directory> <name>\n")
       os.Exit(1)
   }


   rootDir := os.Args[1] // This should be the 'src' directory
   mapName := os.Args[2] // Name passed in while calling from terminal
   repoMap := make(map[string][]string)
   fset := token.NewFileSet()


   // Walk through the directory and all its subdirectories
   filepath.WalkDir(rootDir, func(path string, d fs.DirEntry, err error) error {
       if err != nil {
           return err
       }


       if !d.IsDir() && strings.HasSuffix(d.Name(), ".go") {
           parsedFile, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
           if err != nil {
               return err
           }


           // Initialize dependencies as an empty slice instead of nil
           dependencies := make(map[string]bool)


           // Analyze AST and find dependencies
           ast.Inspect(parsedFile, func(n ast.Node) bool {
               switch x := n.(type) {
               case *ast.SelectorExpr:
                   if ident, ok := x.X.(*ast.Ident); ok {
                       pkg := ident.Name


                       // Check for local files that may correspond to this identifier
                       filepath.WalkDir(rootDir, func(depPath string, depInfo fs.DirEntry, depErr error) error {
                           if depErr != nil {
                               return depErr
                           }


                           if !depInfo.IsDir() && strings.TrimSuffix(depInfo.Name(), ".go") == pkg {
                               relPath, err := filepath.Rel(rootDir, strings.TrimPrefix(depPath, rootDir+string(os.PathSeparator)))
                               if err == nil {
                                   dependencies[relPath] = true
                               }
                           }


                           return nil
                       })
                   }
               }


               return true
           })


           // Convert map keys to a slice
           deps := make([]string, 0) // Initialize as an empty slice
           for dep := range dependencies {
               deps = append(deps, dep)
           }


           fileRelPath, _ := filepath.Rel(rootDir, strings.TrimPrefix(path, rootDir+string(os.PathSeparator)))
           repoMap[fileRelPath] = deps
       }


       return nil
   })


   output := DependencyMap{Name: mapName, RepoName: repoMap}
   result, err := json.Marshal(output) // Change back to Marshal for single-line JSON output
   if err != nil {
       panic(err)
   }


   // Write the result to a file
   file, err := os.Create("output.json")
   if err != nil {
       panic(err)
   }
   defer file.Close()


   _, err = file.Write(result)
   if err != nil {
       panic(err)
   }
}
