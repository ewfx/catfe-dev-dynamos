package com.example.bankapp;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.data.jpa.repository.JpaRepository;
import jakarta.persistence.*;
import java.math.BigDecimal;
import java.util.Optional;

@SpringBootApplication
public class BankAppApplication {
    public static void main(String[] args) {
        SpringApplication.run(BankAppApplication.class, args);
    }
}

@RestController
@RequestMapping("/account")
class AccountController {
    @Autowired
    private AccountService accountService;

    @PostMapping("/credit")
    public Account credit(@RequestParam Long accountId, @RequestParam BigDecimal amount) {
        return accountService.credit(accountId, amount);
    }

    @PostMapping("/debit")
    public Account debit(@RequestParam Long accountId, @RequestParam BigDecimal amount) {
        return accountService.debit(accountId, amount);
    }

    @GetMapping("/balance")
    public BigDecimal getBalance(@RequestParam Long accountId) {
        return accountService.getBalance(accountId);
    }
}

@Service
class AccountService {
    @Autowired
    private AccountRepository accountRepository;

    public Account credit(Long accountId, BigDecimal amount) {
        Account account = accountRepository.findById(accountId).orElseThrow(() -> new RuntimeException("Account not found"));
        account.setBalance(account.getBalance().add(amount));
        return accountRepository.save(account);
    }

    public Account debit(Long accountId, BigDecimal amount) {
        Account account = accountRepository.findById(accountId).orElseThrow(() -> new RuntimeException("Account not found"));
        if (account.getBalance().compareTo(amount) < 0) {
            throw new RuntimeException("Insufficient funds");
        }
        account.setBalance(account.getBalance().subtract(amount));
        return accountRepository.save(account);
    }

    public BigDecimal getBalance(Long accountId) {
        return accountRepository.findById(accountId).map(Account::getBalance).orElseThrow(() -> new RuntimeException("Account not found"));
    }
}

@Entity
class Account {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private BigDecimal balance = BigDecimal.ZERO;

    public Long getId() {
        return id;
    }

    public BigDecimal getBalance() {
        return balance;
    }

    public void setBalance(BigDecimal balance) {
        this.balance = balance;
    }
}

interface AccountRepository extends JpaRepository<Account, Long> {}
