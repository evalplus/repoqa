package edu.cs.illinois.repoqa;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.SourceRoot;

public class DepAnalyze {
    public static String getStartPackage(Path srcRootPath, Path filePath) {
        return srcRootPath.relativize(filePath.getParent()).toString().replace(File.separator, ".");
    }

    public static void analyze(String repoPath, String entryPoint, String filePath) {
        Path srcRootPath = Paths.get(repoPath, entryPoint).toAbsolutePath(); // src/main/java
        CombinedTypeSolver combinedTypeSolver = new CombinedTypeSolver();
        combinedTypeSolver.add(new ReflectionTypeSolver());
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(combinedTypeSolver);
        ParserConfiguration parserConfiguration = new ParserConfiguration().setSymbolResolver(symbolSolver).setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_14);
        SourceRoot sourceRoot = new SourceRoot(srcRootPath, parserConfiguration);
        CompilationUnit cu = sourceRoot.parse(getStartPackage(srcRootPath, Paths.get(filePath)), new File(filePath).getName());

        List<Path> depPaths = new ArrayList<>();
        depPaths.addAll(getImportDepPaths(cu, srcRootPath));
        depPaths.addAll(getSamePackageDepPaths(cu, Paths.get(filePath).toAbsolutePath()));

        depPaths = new ArrayList<>(new HashSet<>(depPaths)); // remove duplicates
        Collections.sort(depPaths);
        depPaths.remove(Paths.get(repoPath)); // remove the current file from the dependencies
        depPaths.forEach(p -> System.out.println(Paths.get(repoPath).relativize(p)));
    }

    private static List<Path> getImportDepPaths(CompilationUnit cu, Path srcRootPath) {
        List<Path> depPaths = new ArrayList<>();
        for (ImportDeclaration importDeclaration : cu.getImports()) {
            String importStr = importDeclaration.getNameAsString();
            if (!importDeclaration.isAsterisk()) {
                Path depPath = srcRootPath.resolve(importStr.replace(".", File.separator) + ".java");
                if (depPath.toFile().exists()) {
                    depPaths.add(depPath);
                } else {
                    Path possibleJavaPath = srcRootPath.resolve(importStr.substring(0, importStr.lastIndexOf(".")).replace(".", File.separator) + ".java");
                    if (possibleJavaPath.toFile().exists()) {
                        // this indicates that the import is like "import static com.example.Main.main;"
                        depPaths.add(possibleJavaPath);
                    }
                }
            }
            else {
                Path depDirPath = srcRootPath.resolve(importStr.replace(".", File.separator));
                Path possibleJavaPath = srcRootPath.resolve(importStr.substring(0, importStr.length()).replace(".", File.separator) + ".java");
                if (possibleJavaPath.toFile().exists()) {
                    // this indicates that the import is like "import com.example.Main.*;"
                    depPaths.add(possibleJavaPath);
                    continue;
                }
                if (depDirPath.toFile().exists()) {
                    File[] files = depDirPath.toFile().listFiles();
                    for (File file : files) {
                        if (file.isFile() && file.getName().endsWith(".java")) {
                            depPaths.add(file.toPath());
                        }
                    }
                }
            }
        }

        return depPaths;
    }

    private static List<Path> getSamePackageDepPaths(CompilationUnit cu, Path filePath) {
        List<Path> depPaths = new ArrayList<>();
        List<String> siblingClassSimpleNameList = new ArrayList<>();
        for (File siblingFile : filePath.getParent().toFile().listFiles()) {
            if (siblingFile.getAbsolutePath().endsWith(".java") && !siblingFile.getAbsolutePath().equals(filePath.toFile().getAbsolutePath())) {
                siblingClassSimpleNameList.add(siblingFile.getName().substring(0, siblingFile.getName().length() - 5));
            }
        }

        // check all identifiers in the current file
        for (NameExpr name : cu.findAll(NameExpr.class)) {
            String identifier = name.getNameAsString();
            if (siblingClassSimpleNameList.contains(identifier)) {
                Path depPath = filePath.getParent().resolve(identifier + ".java");
                if (depPath.toFile().exists()) {
                    depPaths.add(depPath);
                }
            }
        }

        return depPaths;
    }

    public static void main(String[] args) {
        String repoPath = args[0];
        String entryPoint = args[1]; // entry point is relative to the repoPath
        String filePath = args[2];
        analyze(repoPath, entryPoint, filePath);
    }
}
